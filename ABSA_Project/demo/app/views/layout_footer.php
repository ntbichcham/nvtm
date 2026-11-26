</main>

        <footer class="app-footer">
            <span>Demo học phần · Dữ liệu UIT-ViSFD · Kết quả suy luận từ checkpoint PhoBERT đã huấn luyện</span>
        </footer>
    </div>
</div>

<script src="<?= e(asset('vendor/bootstrap/bootstrap.bundle.min.js')) ?>"></script>
<script src="<?= e(asset('js/app.js')) ?>"></script>
<?php if (!empty($withCharts)): ?>
    <script src="<?= e(asset('vendor/chart.js')) ?>"></script>
    <script src="<?= e(asset('js/dashboard.js')) ?>"></script>
<?php endif; ?>
</body>
</html>